#!/bin/bash
# Auto Trader Runner Script
# Runs every 5 minutes during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)

cd /Users/jakobmccleary/Desktop/Stock\ Predictor

while true; do
    # Get current hour (ET timezone)
    HOUR=$(TZ=America/New_York date +%H)
    MIN=$(TZ=America/New_York date +%M)
    DAY=$(TZ=America/New_York date +%u)  # 1=Mon, 7=Sun
    
    # Only trade Mon-Fri (1-5) between 9:30 and 16:00 ET
    if [ "$DAY" -ge 1 ] && [ "$DAY" -le 5 ]; then
        if [ "$HOUR" -ge 9 ] && [ "$HOUR" -lt 16 ]; then
            # Skip before 9:30
            if [ "$HOUR" -eq 9 ] && [ "$MIN" -lt 30 ]; then
                echo "$(date) - Market not open yet (before 9:30 ET)"
            else
                echo "$(date) - Running auto trader..."
                python3 auto_paper_trade.py
            fi
        else
            echo "$(date) - Market closed (hour=$HOUR)"
        fi
    else
        echo "$(date) - Weekend (day=$DAY)"
    fi
    
    # Sleep 5 minutes
    sleep 300
done
