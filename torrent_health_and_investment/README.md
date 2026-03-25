# SwarmHealth - Creative Commons Torrent Health Checker

SwarmHealth monitors the health of Creative Commons licensed torrents discovered through the IPv8 peer network. It tracks seeding levels, peer counts, and calculates growth metrics, displayed in a live GUI dashboard.

## Features

- **Health Monitoring**: Tracks seeders, leechers, and total peer counts for torrents
- **Metrics Calculation**:
  - **Growth**: Percentage change in peer count over time
  - **Shrink**: Measures how much a swarm is shrinking
  - **Exploding Estimator**: Score (0-100) indicating rapid swarm growth
- **Creative Commons Filtering**: Only monitors torrents with Creative Commons licenses
- **Retry Logic**: Automatically retries health checks if DHT returns empty results
- **GUI Dashboard**: Real-time graphical interface with sortable metrics table
- **IPv8 Network**: Discovers torrents via the Liberation Community gossip protocol
- **Seedbox Fleet**: Shows connected nodes with uptime, disk usage, BTC balance, and region

## Installation

1. **Install Python dependencies:**

   ```bash
   python -m pip install --upgrade -r requirements.txt
   ```
   
## Usage

```bash
python -m healthchecker
```

Optionally specify a custom IPv8 key file:

```bash
python -m healthchecker --key-file /path/to/key.pem
```

On startup, the service connects to the IPv8 network, waits for peers to propagate torrent content, then opens the GUI dashboard. The default key file is `liberation_key.pem` in the working directory.

## GUI Dashboard

- **Statistics bar**: Total entries, Healthy, No Peers, Exploding counts
- **Metrics table**: Infohash, seeders, leechers, total peers, growth%, shrink%, exploding score, status, last checked — all columns sortable
- **Seedbox Fleet**: Connected IPv8 nodes with name, IP, git commit, uptime, disk usage, BTC balance, region, last seen
- **Log window**: Timestamped operational log
- **Controls**: Start/Stop health checker, manual Refresh
- Auto-refreshes every 30 seconds

## Metrics Explained

### Seeders

Number of peers that have the complete file and are uploading.

### Leechers

Number of peers that are downloading and don't have the complete file yet.

### Total Peers

Sum of seeders and leechers.

### Growth (%)

Percentage change in peer count compared to the previous check.

- Positive: Swarm is growing
- Negative: Swarm is shrinking
- Example: `+15.5%` means 15.5% more peers than last check

### Shrink (%)

Measures how much a swarm is shrinking (inverse of negative growth).

- Only shows positive values when shrinking
- Example: `10.2%` means the swarm shrunk by 10.2%

### Exploding Estimator (0-100)

A composite score indicating rapid swarm growth:

- **0-30**: Normal growth
- **30-50**: Moderate growth
- **50-70**: High growth
- **70-100**: Explosive growth

The score considers:

- Recent growth rate
- Acceleration (rate of change)
- Number of samples
- Current peer count

## Database

Health check results are stored in `dht_health.db` (SQLite database). The database includes:

- Historical peer counts
- Seeder/leecher counts
- Calculated metrics (growth, shrink, exploding)
- Timestamps for each check
- Source URLs and license information
- Content received from IPv8 peers

## Retry Logic

If a DHT query returns no peers:

1. The system waits 60 seconds
2. Retries the health check (up to 3 attempts)
3. If still no peers after retries, records as "no_peers"

## File Structure

```
torrent_health_and_investment/
├── healthchecker/
│   ├── __init__.py
│   ├── __main__.py              # Entry point
│   ├── client.py                # DHT client and torrent connection
│   ├── db.py                    # Database operations
│   ├── gui.py                   # GUI dashboard
│   ├── liberation_community.py  # IPv8 community protocol
│   ├── liberation_service.py    # IPv8 service lifecycle
│   ├── metrics.py               # Metrics calculations
│   └── sampler.py               # Health checker core logic
├── event_log_server.py          # Mycelium fleet event log server
├── requirements.txt
├── README.md
├── liberation_key.pem           # IPv8 peer identity key
├── dht_health.db                # SQLite health check database
```

---

**⚠️ Important**: This tool is for monitoring Creative Commons licensed torrents only. Do not use for copyrighted content.
