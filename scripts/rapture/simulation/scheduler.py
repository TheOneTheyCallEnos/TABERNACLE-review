"""
LVS Scheduler
=============
Auto-run setup for Mac Mini.

Creates a launchd plist to run daily scans automatically.
"""

import os
import sys
from pathlib import Path


PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lvs.navigator</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>StartCalendarInterval</key>
    <array>
        <!-- 9:30 AM EST (market open) -->
        <dict>
            <key>Hour</key>
            <integer>9</integer>
            <key>Minute</key>
            <integer>30</integer>
        </dict>
        <!-- 4:00 PM EST (market close) -->
        <dict>
            <key>Hour</key>
            <integer>16</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
    </array>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/lvs_output.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/lvs_error.log</string>
    
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""

PAPER_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lvs.papertrader</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>StartCalendarInterval</key>
    <array>
        <!-- Run every hour during market hours -->
        <dict>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <dict>
            <key>Hour</key>
            <integer>11</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <dict>
            <key>Hour</key>
            <integer>12</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <dict>
            <key>Hour</key>
            <integer>13</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <dict>
            <key>Hour</key>
            <integer>14</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <dict>
            <key>Hour</key>
            <integer>15</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
    </array>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/paper_output.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/paper_error.log</string>
    
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def get_python_path():
    """Get the current Python interpreter path."""
    return sys.executable


def setup_scheduler(project_dir: str = None):
    """
    Set up launchd scheduler for Mac.
    """
    if project_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    project_dir = Path(project_dir).resolve()
    python_path = get_python_path()
    
    # Create logs directory
    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    launch_agents_dir.mkdir(exist_ok=True)
    
    # Daily scan plist
    scan_plist = PLIST_TEMPLATE.format(
        python_path=python_path,
        script_path=str(project_dir / "run.py"),
        working_dir=str(project_dir),
        log_dir=str(log_dir)
    )
    
    scan_plist_path = launch_agents_dir / "com.lvs.navigator.plist"
    with open(scan_plist_path, 'w') as f:
        f.write(scan_plist)
    
    print(f"✅ Created: {scan_plist_path}")
    
    # Paper trader plist
    paper_plist = PAPER_PLIST_TEMPLATE.format(
        python_path=python_path,
        script_path=str(project_dir / "simulation" / "paper_trader.py"),
        working_dir=str(project_dir),
        log_dir=str(log_dir)
    )
    
    paper_plist_path = launch_agents_dir / "com.lvs.papertrader.plist"
    with open(paper_plist_path, 'w') as f:
        f.write(paper_plist)
    
    print(f"✅ Created: {paper_plist_path}")
    
    print("\n" + "=" * 50)
    print("  SCHEDULER SETUP COMPLETE")
    print("=" * 50)
    print(f"""
To enable the schedulers, run:

  launchctl load {scan_plist_path}
  launchctl load {paper_plist_path}

To disable:

  launchctl unload {scan_plist_path}
  launchctl unload {paper_plist_path}

To run immediately:

  launchctl start com.lvs.navigator
  launchctl start com.lvs.papertrader

Logs are at:
  {log_dir}/lvs_output.log
  {log_dir}/paper_output.log

Schedule:
  - Daily scan: 9:30 AM and 4:00 PM
  - Paper trader: Every hour 10 AM - 3 PM
""")


def remove_scheduler():
    """Remove scheduler plists."""
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    
    plists = [
        launch_agents_dir / "com.lvs.navigator.plist",
        launch_agents_dir / "com.lvs.papertrader.plist"
    ]
    
    for plist in plists:
        if plist.exists():
            # Unload first
            os.system(f"launchctl unload {plist} 2>/dev/null")
            plist.unlink()
            print(f"✅ Removed: {plist}")
        else:
            print(f"⚠️ Not found: {plist}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', action='store_true', help='Remove scheduler')
    parser.add_argument('--dir', type=str, help='Project directory')
    args = parser.parse_args()
    
    if args.remove:
        remove_scheduler()
    else:
        setup_scheduler(args.dir)
