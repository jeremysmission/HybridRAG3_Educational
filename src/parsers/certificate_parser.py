# ============================================================================
# HybridRAG -- Certificate Parser (src/parsers/certificate_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads X.509 digital certificates (.cer, .crt, .pem) and extracts
#   human-readable information: who issued it, who it belongs to,
#   when it expires, what algorithms it uses.
#
# WHY THIS MATTERS:
#   System administrators and cybersecurity teams manage hundreds of
#   certificates. Being able to search "which certificates expire in
#   March?" or "what certificates use SHA-1?" is valuable.
#
# DEPENDENCIES:
#   pip install cryptography  (Apache 2.0 / BSD-3, already in most envs)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


class CertificateParser:
    """
    Extract metadata from X.509 certificates (.cer, .crt, .pem).

    NON-PROGRAMMER NOTE:
      A digital certificate is like a driver's license for a server.
      It says "This server is who it claims to be" and is signed by
      a trusted authority (like a DMV signs your license).
      We extract the "readable" parts: name, issuer, dates, algorithm.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "CertificateParser"}

        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import serialization
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install cryptography"
            )
            return "", details

        try:
            raw = path.read_bytes()
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read file: {e}"
            return "", details

        # Try PEM format first (base64 text), then DER (binary)
        cert = None
        try:
            cert = x509.load_pem_x509_certificate(raw)
            details["format"] = "PEM"
        except Exception:
            try:
                cert = x509.load_der_x509_certificate(raw)
                details["format"] = "DER"
            except Exception as e:
                details["error"] = f"PARSE_ERROR: Not a valid certificate: {e}"
                return "", details

        parts = [f"X.509 Certificate: {path.name}"]

        try:
            parts.append(f"Subject: {cert.subject.rfc4514_string()}")
            parts.append(f"Issuer: {cert.issuer.rfc4514_string()}")
            parts.append(f"Serial: {cert.serial_number}")
            parts.append(f"Not Before: {cert.not_valid_before_utc}")
            parts.append(f"Not After: {cert.not_valid_after_utc}")
            parts.append(f"Signature Algorithm: {cert.signature_algorithm_oid.dotted_string}")
            parts.append(f"Version: {cert.version}")

            # Subject Alternative Names (if present)
            try:
                san = cert.extensions.get_extension_for_class(
                    x509.SubjectAlternativeName
                )
                names = san.value.get_values_for_type(x509.DNSName)
                if names:
                    parts.append(f"SAN DNS Names: {', '.join(names)}")
            except x509.ExtensionNotFound:
                pass

        except Exception as e:
            details["error"] = f"PARSE_ERROR: {e}"

        full = "\n".join(parts).strip()
        details["total_len"] = len(full)
        return full, details
