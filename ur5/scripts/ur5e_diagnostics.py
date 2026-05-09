#!/usr/bin/env python3
"""UR5e SSH diagnostic script - all read-only commands."""

import paramiko
import sys

HOST = "192.10.0.11"
USER = "root"
PASSWORDS = ["", "easybot"]

def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for pwd in PASSWORDS:
        try:
            client.connect(HOST, username=USER, password=pwd, timeout=10)
            print(f"[OK] Connected with password: {'(empty)' if pwd == '' else repr(pwd)}")
            return client
        except paramiko.AuthenticationException:
            print(f"[FAIL] Password {'(empty)' if pwd == '' else repr(pwd)} rejected")
            continue
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            sys.exit(1)
    print("[FATAL] All passwords failed")
    sys.exit(1)

def run(client, cmd, label=None):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"$ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print(f"[stderr] {err.rstrip()}")
    print()
    return out

def main():
    client = ssh_connect()

    # Step 2: Boot & Kernel Logs
    run(client, "dmesg | head -200", "STEP 2a: Kernel boot log (first 200 lines)")
    run(client, 'dmesg | grep -iE "error|fail|timeout|warn|delay|reset|retry"',
        "STEP 2b: Kernel errors/warnings")
    run(client, "systemd-analyze 2>/dev/null || echo 'systemd-analyze not available'",
        "STEP 2c: Boot timing")
    run(client, "systemd-analyze blame 2>/dev/null | head -30",
        "STEP 2d: Boot blame (top 30)")
    run(client, "journalctl -b -p err --no-pager 2>/dev/null | head -50",
        "STEP 2e: Journal errors from current boot")
    run(client, "journalctl -b --no-pager 2>/dev/null | head -100",
        "STEP 2f: Journal first 100 lines")

    # Step 3: Hardware Info
    run(client, "dmidecode -t bios 2>/dev/null || echo 'dmidecode not available'",
        "STEP 3a: BIOS info")
    run(client, "dmidecode -t system 2>/dev/null", "STEP 3b: System info")
    run(client, "dmidecode -t baseboard 2>/dev/null", "STEP 3c: Baseboard info")
    run(client, "date", "STEP 3d: System date")
    run(client, "hwclock --show 2>/dev/null || echo 'hwclock not available'",
        "STEP 3e: Hardware RTC clock")
    run(client, "dmidecode -t memory 2>/dev/null", "STEP 3f: Memory info")
    run(client, "free -h", "STEP 3g: Memory usage")

    # Step 4: Storage Health
    run(client, "lsblk", "STEP 4a: Block devices")
    run(client, "df -h", "STEP 4b: Disk usage")
    run(client, "mount", "STEP 4c: Mount points")
    run(client, "cat /sys/block/mmcblk0/device/life_time 2>/dev/null || echo 'No eMMC life_time'",
        "STEP 4d: eMMC/SD wear indicator")
    run(client, "cat /sys/block/mmcblk0/device/name 2>/dev/null || echo 'No eMMC name'",
        "STEP 4e: eMMC/SD device name")
    run(client, 'dmesg | grep -i "mmc\\|sd\\|block"', "STEP 4f: Storage kernel messages")
    run(client, 'dmesg | grep -iE "ext4|filesystem|corrupt|readonly|I/O error"',
        "STEP 4g: Filesystem errors")

    # Step 5: PCIe / USB
    run(client, "lspci 2>/dev/null || echo 'lspci not available'", "STEP 5a: PCIe devices")
    run(client, 'lspci -vv 2>&1 | grep -iE "error|correctable|fatal"',
        "STEP 5b: PCIe errors")
    run(client, "lsusb 2>/dev/null || echo 'lsusb not available'", "STEP 5c: USB devices")
    run(client, 'dmesg | grep -iE "pci|usb|enumerat"', "STEP 5d: PCI/USB kernel messages")

    # Step 6: UR-Specific Logs
    run(client, "ls -la /root/ur-*/ 2>/dev/null || echo 'No /root/ur-* dirs'",
        "STEP 6a: UR directories")
    run(client, "ls -la /programs/ 2>/dev/null || echo 'No /programs/ dir'",
        "STEP 6b: Programs directory")
    run(client, 'find / -name "*.log" -path "*ur*" 2>/dev/null | head -20',
        "STEP 6c: UR log files")
    run(client, 'find / -name "polyscope*" 2>/dev/null | head -10',
        "STEP 6d: Polyscope files")
    run(client, "cat /tmp/ur_* 2>/dev/null || echo 'No /tmp/ur_* files'",
        "STEP 6e: UR temp files")
    run(client, "ls -la /root/.urcontrol/ 2>/dev/null || echo 'No .urcontrol dir'",
        "STEP 6f: URControl directory")

    # Step 7: Thermal / Power
    run(client, "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null || echo 'No thermal zones'",
        "STEP 7a: CPU temperature")
    run(client, 'dmesg | grep -iE "power|voltage|thermal|overheat|throttl"',
        "STEP 7b: Power/thermal kernel messages")
    run(client, "uptime", "STEP 7c: Uptime")

    client.close()
    print("\n" + "="*60)
    print("  DIAGNOSTICS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
