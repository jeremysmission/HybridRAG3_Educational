# ============================================================================
# HybridRAG -- PCAP Network Capture Parser (src/parsers/pcap_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads network packet capture files (.pcap, .pcapng) and extracts
#   summary metadata: packet count, protocol breakdown, IP addresses,
#   time range, and any cleartext content.
#
# WHY THIS MATTERS:
#   Cybersecurity analysts store packet captures for incident analysis.
#   Being able to search "captures involving IP 10.0.1.50" or "captures
#   with DNS traffic" helps quickly locate relevant evidence.
#
# DEPENDENCIES:
#   pip install dpkt  (BSD-3 license)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple


class PcapParser:
    """
    Extract summary metadata from PCAP/PCAPNG network captures.

    NON-PROGRAMMER NOTE:
      A .pcap file contains recorded network traffic -- like a
      surveillance tape for network packets. We don't extract the
      full raw traffic (too large), but we summarize what's in it:
      how many packets, which IP addresses, which protocols, time range.
    """

    MAX_PACKETS = 10000  # Analyze up to 10K packets

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "PcapParser"}

        try:
            import dpkt
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install dpkt"
            )
            return "", details

        try:
            with open(str(path), "rb") as f:
                try:
                    pcap = dpkt.pcap.Reader(f)
                except Exception:
                    f.seek(0)
                    try:
                        pcap = dpkt.pcapng.Reader(f)
                    except Exception as e:
                        details["error"] = f"RUNTIME_ERROR: Not a valid pcap: {e}"
                        return "", details

                return self._analyze(pcap, path, details)
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: {e}"
            return "", details

    def _analyze(self, pcap, path, details) -> Tuple[str, Dict[str, Any]]:
        import dpkt
        from collections import Counter
        import socket

        parts: List[str] = [f"Network Capture: {path.name}"]
        pkt_count = 0
        protos: Counter = Counter()
        src_ips: Counter = Counter()
        dst_ips: Counter = Counter()
        ts_first = None
        ts_last = None

        for ts, buf in pcap:
            if pkt_count >= self.MAX_PACKETS:
                break
            pkt_count += 1
            if ts_first is None:
                ts_first = ts
            ts_last = ts

            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data
                    src = socket.inet_ntoa(ip.src)
                    dst = socket.inet_ntoa(ip.dst)
                    src_ips[src] += 1
                    dst_ips[dst] += 1

                    if isinstance(ip.data, dpkt.tcp.TCP):
                        protos["TCP"] += 1
                    elif isinstance(ip.data, dpkt.udp.UDP):
                        protos["UDP"] += 1
                    elif isinstance(ip.data, dpkt.icmp.ICMP):
                        protos["ICMP"] += 1
                    else:
                        protos["Other IP"] += 1
                else:
                    protos["Non-IP"] += 1
            except Exception:
                protos["Malformed"] += 1

        parts.append(f"Packets analyzed: {pkt_count:,}")

        if ts_first and ts_last:
            from datetime import datetime, timezone
            t1 = datetime.fromtimestamp(ts_first, tz=timezone.utc)
            t2 = datetime.fromtimestamp(ts_last, tz=timezone.utc)
            parts.append(f"Time range: {t1.isoformat()} to {t2.isoformat()}")

        if protos:
            parts.append("Protocols: " + ", ".join(
                f"{p}={c}" for p, c in protos.most_common(10)
            ))

        if src_ips:
            top_src = src_ips.most_common(10)
            parts.append("Top source IPs: " + ", ".join(
                f"{ip}({c})" for ip, c in top_src
            ))

        if dst_ips:
            top_dst = dst_ips.most_common(10)
            parts.append("Top dest IPs: " + ", ".join(
                f"{ip}({c})" for ip, c in top_dst
            ))

        full = "\n".join(parts).strip()
        details["total_len"] = len(full)
        details["packets"] = pkt_count
        return full, details
