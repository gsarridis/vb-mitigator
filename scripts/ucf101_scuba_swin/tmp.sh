# Replace 'username' with the user and 12345 with the job ID threshold
USER="isarridis"
THRESHOLD=142483

# List job IDs greater than the threshold and cancel them
squeue -u $USER -h -o "%i" | while read jid; do
    if [ "$jid" -gt "$THRESHOLD" ]; then
        echo "Cancelling job $jid"
        scancel $jid
    fi
done