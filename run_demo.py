"""
CLI Launcher for the Diabetes Multi-Modal Insulin Prediction Demo Dashboard.

Usage:
    python run_demo.py
    python run_demo.py --port 8501
    python run_demo.py --version
    python run_demo.py --help
"""

import argparse
import os
import sys
import socket
import subprocess


def find_available_port(start_port: int, host: str = "localhost") -> int:
    """Find the next available TCP port if the requested port is busy."""
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1
    return start_port


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Interactive Demonstration Dashboard for Module 1 and Module 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0 (Diabetes Insulin Prediction Demo Module - powered by standard argparse)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Preferred port to run the Streamlit demo server on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address to bind the server to",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open browser on startup",
    )

    args = parser.parse_args()

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "app.py")
    if not os.path.exists(app_path):
        print(f"Error: Demo application file not found at: {app_path}", file=sys.stderr)
        sys.exit(1)

    # Automatically resolve port conflicts
    port = find_available_port(args.port, args.host)
    if port != args.port:
        print(f"[!] Note: Port {args.port} was busy. Automatically switched to available port {port}.")

    print("\n" + "=" * 60)
    print("STARTING DIABETES MULTI-MODAL DEMONSTRATION DASHBOARD")
    print("=" * 60)
    print(f"Host:    {args.host}")
    print(f"Port:    {port}")
    print(f"URL:     http://{args.host}:{port}")
    print("Target:  Module 1 (Data Preprocessing) & Module 2 (Knowledge Graph)")
    print("To stop: Press Ctrl+C in this terminal")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        f"--server.address={args.host}",
        "--browser.gatherUsageStats=false",
    ]
    if args.no_browser:
        cmd.append("--server.headless=true")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDemonstration server stopped successfully.")


if __name__ == "__main__":
    main()
