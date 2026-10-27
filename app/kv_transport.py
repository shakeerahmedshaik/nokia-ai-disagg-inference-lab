"""KV-cache transports between the prefill worker (rank 0) and the decode worker (rank 1).

RoceTransport pushes the slab with RDMA WRITE over RC queue pairs (libibverbs, Soft-RoCE `rxe`), the
"WRITE (push)" example of Learning Byte 04:

  * decode allocates and registers the destination slab and publishes (addr, rkey) in the TCP store,
  * prefill pushes byte ranges of its own registered slab with RDMA WRITE WITH IMMEDIATE,
  * each range is striped across `channels` RC QPs, and every QP has its own UDP source port, so the
    fabric can ECMP-hash the channels onto different spines,
  * the immediate (the layer index) tells decode that a range landed; decode counts completions.

TcpTransport sends the same slab over one TCP connection, for comparison.

Only bootstrap and metadata use the TCP store, as NCCL and NIXL bootstrap over sockets.
"""
import ctypes
import glob
import json
import random
import socket
import time

import numpy as np

ROCE_DSCP = 26
ECT0 = 0b10
PORT = 1
TCP_DATA_PORT = 29600


def stripes(offset, nbytes, parts):
    """Split [offset, offset+nbytes) into `parts` contiguous ranges (last ones may be shorter or empty)."""
    size = -(-nbytes // parts)
    out = []
    for i in range(parts):
        lo = offset + i * size
        n = max(0, min(size, offset + nbytes - lo))
        if n:
            out.append((i, lo, n))
    return out


def roce_v2_ipv4_gid_index(dev):
    base = f"/sys/class/infiniband/{dev}/ports/{PORT}"
    for path in sorted(glob.glob(f"{base}/gids/*"), key=lambda p: int(p.rsplit("/", 1)[1])):
        idx = int(path.rsplit("/", 1)[1])
        try:
            gid = open(path).read().strip()
            gtype = open(f"{base}/gid_attrs/types/{idx}").read().strip()
        except OSError:
            continue
        if gid.startswith("0000:0000:0000:0000:0000:ffff") and "v2" in gtype:
            return idx
    raise RuntimeError(f"no RoCE v2 IPv4 GID on {dev}")


class RoceTransport:
    def __init__(self, store, role, dev, channels=2, tclass=(ROCE_DSCP << 2) | ECT0, log=print):
        import pyverbs.enums as e
        from pyverbs.addr import GID, AHAttr, GlobalRoute
        from pyverbs.cq import CQ
        from pyverbs.device import Context
        from pyverbs.pd import PD
        from pyverbs.qp import QP, QPAttr, QPCap, QPInitAttr

        self.e, self.store, self.role, self.channels, self.log = e, store, role, channels, log
        rank = 0 if role == "prefill" else 1
        self.ctx = Context(name=dev)
        self.pd = PD(self.ctx)
        self.cq = CQ(self.ctx, 4096)
        self.access = e.IBV_ACCESS_LOCAL_WRITE | e.IBV_ACCESS_REMOTE_WRITE
        self.imm_sink = self._mr(64, e.IBV_ACCESS_LOCAL_WRITE)
        self.gid_index = roce_v2_ipv4_gid_index(dev)
        gid = self.ctx.query_gid(PORT, self.gid_index)

        cap = QPCap(max_send_wr=512, max_recv_wr=1024, max_send_sge=1, max_recv_sge=1)
        self.qps = [QP(self.pd, QPInitAttr(qp_type=e.IBV_QPT_RC, scq=self.cq, rcq=self.cq, cap=cap))
                    for _ in range(channels)]
        psn = [random.randrange(1 << 24) for _ in range(channels)]
        me = dict(gid=str(gid), qpn=[q.qp_num for q in self.qps], psn=psn)
        store.set(f"roce/{rank}", json.dumps(me))
        peer = json.loads(store.get(f"roce/{1 - rank}"))

        for i, qp in enumerate(self.qps):
            attr = QPAttr(port_num=PORT)
            attr.qp_access_flags = self.access
            qp.to_init(attr)
            gr = GlobalRoute(dgid=GID(peer["gid"]), sgid_index=self.gid_index, hop_limit=64, traffic_class=tclass)
            attr.ah_attr = AHAttr(port_num=PORT, is_global=1, gr=gr)
            attr.path_mtu = e.IBV_MTU_4096
            attr.dest_qp_num = peer["qpn"][i]
            attr.rq_psn = peer["psn"][i]
            attr.max_dest_rd_atomic = 1
            attr.min_rnr_timer = 12
            qp.to_rtr(attr)
            attr.sq_psn = psn[i]
            attr.timeout = 14
            attr.retry_cnt = 7
            attr.rnr_retry = 7
            attr.max_rd_atomic = 1
            qp.to_rts(attr)
            for _ in range(256):
                qp.post_recv(self._recv_wr())
        self.qpn_to_ch = {q.qp_num: i for i, q in enumerate(self.qps)}
        store.set(f"roce/ready/{rank}", "1")
        store.get(f"roce/ready/{1 - rank}")
        self.slab_mr = None
        self.remote = None
        self.inflight = 0
        self.wr_seq = 0
        log(f"roce dev={dev} gid_index={self.gid_index} gid={gid} qpn={me['qpn']} peer_gid={peer['gid']} "
            f"peer_qpn={peer['qpn']} channels={channels} tclass=0x{tclass:02x} (DSCP {tclass >> 2}, ECN {tclass & 3})")

    def _mr(self, nbytes, access):
        from pyverbs.mr import MR
        return MR(self.pd, nbytes, access)

    def _recv_wr(self):
        from pyverbs.wr import SGE, RecvWR
        return RecvWR(num_sge=1, sg=[SGE(self.imm_sink.buf, 64, self.imm_sink.lkey)])

    def open_request(self, rid, nbytes):
        """Allocate and register the slab. Decode publishes (addr, rkey) for the sender."""
        if self.slab_mr is not None:
            self.slab_mr.close()
        self.slab_mr = self._mr(nbytes, self.access)
        self.remote = None
        if self.role == "decode":
            self.store.set(f"mr/{rid}", json.dumps([self.slab_mr.buf, self.slab_mr.rkey]))
        raw = (ctypes.c_ubyte * nbytes).from_address(self.slab_mr.buf)
        return np.frombuffer(raw, dtype=np.uint8)

    def push(self, rid, offset, nbytes, tag):
        """RDMA WRITE WITH IMM of slab[offset:offset+nbytes], striped over the channels. imm_data = tag."""
        from pyverbs.wr import SGE, SendWR
        if self.remote is None:
            self.remote = json.loads(self.store.get(f"mr/{rid}"))
        raddr, rkey = self.remote
        for ch, lo, n in stripes(offset, nbytes, self.channels):
            self.wr_seq += 1
            wr = SendWR(wr_id=self.wr_seq, opcode=self.e.IBV_WR_RDMA_WRITE_WITH_IMM, num_sge=1,
                        sg=[SGE(self.slab_mr.buf + lo, n, self.slab_mr.lkey)],
                        send_flags=self.e.IBV_SEND_SIGNALED)
            wr.imm_data = tag
            wr.set_wr_rdma(rkey, raddr + lo)
            self.qps[ch].post_send(wr)
            self.inflight += 1

    def _poll(self, on_recv=None):
        n, wcs = self.cq.poll(num_entries=64)
        for wc in wcs[:n]:
            if wc.status != self.e.IBV_WC_SUCCESS:
                raise RuntimeError(f"WC error status={wc.status} opcode={wc.opcode} qpn={wc.qp_num}")
            if wc.opcode == self.e.IBV_WC_RDMA_WRITE:
                self.inflight -= 1
            elif wc.opcode == self.e.IBV_WC_RECV_RDMA_WITH_IMM:
                self.qps[self.qpn_to_ch[wc.qp_num]].post_recv(self._recv_wr())
                if on_recv:
                    on_recv(wc.imm_data)
        return n

    def flush(self, timeout=300.0):
        """Sender: wait until every WRITE is acknowledged by the responder (RC ACK)."""
        deadline = time.monotonic() + timeout
        while self.inflight:
            if not self._poll() and time.monotonic() > deadline:
                raise TimeoutError(f"{self.inflight} WRITEs unacknowledged")

    def wait_data(self, expected, timeout=300.0):
        """Receiver: wait for `expected` WRITE-with-immediate completions. Returns arrival timestamps."""
        got, first, per_tag = 0, None, {}

        def on_recv(tag):
            nonlocal got, first
            got += 1
            first = first or time.time()
            per_tag[tag] = per_tag.get(tag, 0) + 1

        deadline = time.monotonic() + timeout
        while got < expected:
            if not self._poll(on_recv) and time.monotonic() > deadline:
                raise TimeoutError(f"received {got}/{expected} completions")
        return {"t_first": first, "t_last": time.time(), "messages": got, "tags": len(per_tag)}

    def close_request(self):
        pass


class TcpTransport:
    def __init__(self, store, role, bind_addr="10.10.2.2", peer_addr="10.10.2.2", log=print):
        self.store, self.role, self.log = store, role, log
        self.slab = None
        if role == "decode":
            srv = socket.socket()
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((bind_addr, TCP_DATA_PORT))
            srv.listen(1)
            store.set("tcp/ready", "1")
            self.sock, peer = srv.accept()
            srv.close()
            log(f"tcp data connection from {peer[0]}:{peer[1]}")
        else:
            store.get("tcp/ready")
            self.sock = socket.create_connection((peer_addr, TCP_DATA_PORT))
            log(f"tcp data connection to {peer_addr}:{TCP_DATA_PORT} from port {self.sock.getsockname()[1]}")
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def open_request(self, rid, nbytes):
        self.slab = np.zeros(nbytes, dtype=np.uint8)
        return self.slab

    def push(self, rid, offset, nbytes, tag):
        self.sock.sendall(memoryview(self.slab)[offset:offset + nbytes])

    def flush(self, timeout=300.0):
        pass

    def wait_data(self, expected_unused, timeout=300.0):
        mv, n, first = memoryview(self.slab), 0, None
        while n < len(self.slab):
            got = self.sock.recv_into(mv[n:], min(len(self.slab) - n, 1 << 20))
            if not got:
                raise ConnectionError("peer closed during KV transfer")
            first = first or time.time()
            n += got
        return {"t_first": first, "t_last": time.time(), "messages": 1, "tags": 1}
