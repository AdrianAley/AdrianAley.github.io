# AleyAnalytics



LINUX
* source activate root
* source activate anaconda27r
* source activate intelPython2
* source activate intelPython3
WINDOWS
* activate root
* activate anacondaDev
* activate intelPython2
* activate intelPython3
BOTH
* conda install -c intel ‘package’
* pip list --outdated | cut -d ' ' -f1 | xargs -n1 pip install -U
* conda update --all
* conda config --add channels ‘channel’
* conda create -n ‘name’ intelpython2_full python=2
* conda create -n ‘name’ intelpython3_full python=3
* conda install ‘package’ -c ‘channel’ --no-update-deps
* conda update -c defaults --override-channels –all
* conda update -c intel --override-channels --all
* conda clean --all
* conda create --name ‘name’ python=’version’
* conda env list
* conda create --clone ‘name’ --name ‘new-name’
* conda list
* conda list --revisions
* which -a python
* python --version
* anaconda-navigator --reset
* spyder --reset
* https://docs.anaconda.com/anaconda/packages/pkg-docs

	System Services (Ubuntu)
* systemctl start ‘service’
* systemctl stop ‘service’
* systemctl restart ‘service’
* systemctl enable ‘service’
* systemctl disable ‘service’
* systemctl daemon-reload
* systemctl restart-failed
* service --status-all

	System Maintenance (Ubuntu)
* sudo -i
* sudo apt-get install ‘package’
* sudo apt-get upgrade
* sudo apt-fast update
* sudo apt autoremove
* sudo apt autoclean
* sudo apt-get update --fix-missing
* sudo apt-get --fix-broken install
* sudo rm /var/lib/apt/lists/* -vf  
* sudo apt-get clean
* sudo dpkg --configure -a
* sudo apt-get full-upgrade
* sudo apt dist-upgrade
* sudo apt install -f
* sudo updatedb
* sudo aptitude autoclean
* uname -r
* dpkg --list | grep linux-image
* sudo apt-get purge linux-image-X.X.X-X-generic (X.X.X-X is the kernel to be removed)
* sudo update-grub2

	App Icon Maintenance (Ubuntu)
* sudo update-icon-caches /usr/share/icons/*
* sudo gtk-update-icon-cache
* xprop WM_CLASS

	Steps to Fix MySQL Error in Ubuntu
(Can't connect to local MySQL server through socket ‘/var/run/mysqld/mysqld.sock’)
1. sudo service mysql start OR /etc/init.d/mysql start
2. ps -A|grep mysql
3. sudo pkill mysql
4. ps -A|grep mysqld
5. sudo pkill mysqld
6. sudo service mysql restart
7. mysql -u root -p

	Linux Subsystem on Windows (Ubuntu)
* export DISPLAY=0.0 (use xming server to view ubuntu items)

	Administrator Commands (Windows)
POWERSHELL
* DISM.exe /Online /Cleanup-image /Restorehealth
* Optimize-Volume -DriveLetter C -Analyze -Verbose
* Optimize-Volume -DriveLetter C -Defrag -Verbose
COMMAND PROMPT
* defrag /C /O
* wmic (Windows Management Instrumentation Command-line)
* MOFCOMP %SYSTEMROOT%\System32\WindowsVirtualization.V2.mof (Rebuilds the WMI components for virtualization)
* diskdrive get status (Hard Drive Overall Status)
* bcdedit /set hypervisorlaunchtype auto
* bcdedit /set hypervisorlaunchtype off
* sfc /scannow
* chkdsk <drive>: /f (Fixes Errors Detected)
* /r (Identifies Bad Sectors & Attempts Recovery of Information)
* /I (Performs a Simpler Check of Index Entries)
* /p (Performs an Exhaustive Check of the Current Disk)
* /scan (Run online scan)
* /forceofflinefix (Bypass online repair and queue defects for offline repair. Needs to be used along with /scan)
* /perf (Perform the scan as fast as possible)
* /spotfix (Perform spot repair in offline mode
* /offlinescanandfix Run offline scan and perform fixes
* /sdcclean Garbage collection
Supported by Windows 10 on FAT/FAT32/exFAT volumes only:
* /freeorphanedchains (Free up any orphaned cluster chains)
* /markclean (Mark the volume clean if no corruption is detected)

	Optimize Ethernet Network Settings (Windows)
* Auto Disable Gigabit - Disable
* Auto Disable PCIe - Disable
* Auto Disable PHY - Disable
* Flow Control - Disable
* Green Ethernet - Disable! [Previously set to Enabled]
* Interrupt Moderation - Disable! [Previously set to Enabled]
* IPv4 Checksum Offload - Rx & Tx Enable
* Jumbo Frame - Disable
* Large Send Offload v2 (IPv4) - Enabled [Previously set to Disabled]
* Large Send Offload v2 (IPv6) - Enabled [Previously set to Disabled]
* Network Address - Not Preset
* Priority & VLAN - Priorty & VLAN Disabled
* Receive Buffers - 512 (Max = 512, set to max)
* Receive Side Scaling - Enable (Enabled if you have a multi-core CPU)
* Shutdown Wake-On-Lan - Disable
* Speed & Duplex - 100Mbps Full Duplex [Previously set to Auto Negotiate]
* TCP Checksum Offload (IPv4) - Rx & Tx Enable
* TCP Checksum Offload (IPv6) - Rx & Tx Enable [Previously set to Disabled]
* Transmit Buffers - 128 (Max = 128, set to max)
* UDP Checksum Offload (IPv4) - Rx & Tx Enable
* UDP Checksum Offload (IPv6) - Rx & Tx Enable [Previously set to Disabled]
* Wake-On-Lan Capabilities - None
* WOL & Shutdown Link Speed - 100Mbps First

	R
Start the Rattle GUI
* rattle()

	CLI
* pwd = get path to working directory
* ls = list files in current directory
* rm = remove
* rm -r = remove entire directory contents
* mkdir = make a directory
* cp = copy (ex. cp new_file documents)
* mv = move or rename a file (ex. mv new_file documents (to copy into new directory), mv new_file renamed_file (to rename a file))
* echo = output
* date = print date
* clear = clear screen
* touch = creates a new file


----- SUSE Linux Enterprise Server (SLES) -----


	Zypper Package Management
List repositories
* zypper lr
Add repository
* zypper ar -f <URL> <alias>
Refresh repositories
* zypper ref
Update installed packages
* zypper up
Perform a distribution upgrade
* zypper dup
Package information
* zypper if <package name>
Package search
* zypper se <package, pattern or dependancy name>
Which package owns a file
* zypper se --provides <file path>
List files in package
* $ rpm -ql <package name>

	Network
View network interfaces
* $ ip a
* $ iwconfig
Show routes
* $ ip ru; ip route show table all
Show open TCP/UDP ports
* ss -anptu
Show all open ports
* ss -anp
Test host availability
* $ ping hostname
Change host name
* hostnamectl set-hostname machine.network.name

	Services
List all services
* systemctl list-units --type service
Service status
* systemctl status <service name>
Start/Stop/Restart service
* systemctl start <service name>
* systemctl stop <service name>
* systemctl restart <service name>
Show overriden config files
* systemd-delta
Anaylze boot times
* systemd-analyze blame
* systemd-analyze plot >filename.svg
Show the journal information
* journalctl -u <service name>
* journalctl -f (follow the output of the journal, similar to 'old' tail -f /var/log/messages)
* journalctl -b (only show messages since last boot)
Manage Time and Date
* timedatectl

	CPU & Memory information
View CPU details
* $ lscpu
* $ less /proc/cpuinfo
* $ uname -a
Show running processes
* $ ps -ef
* $ pstree
* $ top -c
Show memory use
* $ less /proc/meminfo
* $ free
Enable/disable swap
* $ swapon -a
* $ swapoff -a
Show all open files & directories
* lsof | less
* lsof | grep -i filename

	File Systems
List disks & partitions
* fdisk -l
* fdisk -l /dev/<h/s>d<a/z>
List mounted file systems
* $ lsblk
* $ findmnt
* $ less /proc/self/mountinfo
Mount a partition
* mount -t <type> <device> <mount point>
Mount a CD/DVD iso image
* mount -t iso9660 -o loop dvd-image.iso <mount point>
Unmount file systems
* umount /dev/<device>
* umount /<mount point>
Inode and disk space usage combined, or output per field type
* df --o -h
* df --output=target,fstype,pcent
space occupied by a file or directory
* du -h
Show all directories occupying more space than 10M
* du -h -t10M

	Accounts
Create user account
* useradd <name>
* -u UID
* -g GID
* -d home directory
* -c full user name
* -s default shell
Delete user account
* userdel <name>
Change user password
* passwd <name>
Modify user account
* usermod <options> <name>

	Build Service
Branch & Checkout a Package
* $ osc bco <source project> <source package>
Commit changes to package
* $ osc commit -m "<comment>"
Submit changed package
* $ osc sr

	Filesystem layout
* /bin – Contains useful commands that are used both user and administrators.
* /boot – This directory contains the boot loader and the Linux kernel.
* /dev – Contains the special device files for all the devices.
* /etc – This directory contains the host-specific configuration files for your system.
* /home – Linux is a multi-user environment, so each user is also assigned a specific directory which is accessible only to them and the system administrator.
* /lib* – Contains shared libraries that are required by system programs.
* /mnt – A generic mount point.
* /opt – Contains third-party software that is not part of openSUSE.
* /proc – Pseudo-file system containing files related to processes and kernel configuration
* /root – Home directory of the user root.
* /run – Files the system creates during its operation, and which do not persist across reboots.
* /sbin – Contains binaries that are essential to the working of the system.
* /srv – Contains site-specific data which is served by this system.
* /sys – Pseudo filesystem containing files pertaining to kernel configuration and system state
* /tmp – Directory to hold temporary files.
* /usr – Directory contains system files and directories shared by all users.
* /var – Contains files to which the system writes data during the course of its operation.
