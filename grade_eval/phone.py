"""Addresses for the phone link: this PC's LAN IP, the ports, and an optional self-signed https certificate."""
import datetime
import ipaddress
import socket
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parents[1] / "data" / "certs"
HTTP_PORT, HTTPS_PORT = 8000, 8443


def lan_ip() -> str:
    """The PC's address on the Wi-Fi/LAN (the interface with the default route); no packet is sent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def cert_files():
    return CERT_DIR / "cert.pem", CERT_DIR / "key.pem"


def ensure_cert() -> tuple[Path, Path]:
    """Self-signed certificate for localhost + this PC's LAN address. Phones only allow the camera on https pages."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    cert, key = cert_files()
    ip = lan_ip()
    stamp = CERT_DIR / "ip.txt"
    if cert.exists() and key.exists() and stamp.exists() and stamp.read_text() == ip:
        return cert, key
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "grade-eval")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                                       x509.IPAddress(ipaddress.ip_address(ip))])
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
         .not_valid_after(now + datetime.timedelta(days=825)).add_extension(san, critical=False).sign(k, hashes.SHA256()))
    cert.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    (CERT_DIR / "ip.txt").write_text(ip)
    return cert, key
