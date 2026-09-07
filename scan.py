import struct

binfile = r'd:\git\ka_zhi_shen\model.bin'
marker = b'@BIN@'

with open(binfile, 'rb') as f:
    content = f.read()

print('total size:', len(content))
cnt = content.count(marker)
print('marker count:', cnt)

positions = []
pos = 0
while True:
    pos = content.find(marker, pos)
    if pos == -1:
        break
    positions.append(pos)
    pos += len(marker)

# each segment: [prev_end, marker_start) is description text, after marker is binary floats
# find binary region length: floats until next description starts
# We use the heuristic: next description starts at some point before next marker.
# Instead, print descriptions with offsets and guess float counts by searching where
# printable text (the next description) begins.
records = []
last_end = 0
for i, mpos in enumerate(positions):
    desc = content[last_end:mpos]
    # description may contain trailing float data from previous bin; strip non-ascii tail heuristically later
    records.append((mpos, desc))
    last_end = mpos + len(marker)

# For each record, find where the *next* description's text begins: the next non-float region.
# We detect start of text for record i+1: scan back from marker i+1 while bytes are part of description.
# Simpler: for each record i, float data starts at mpos+5. Text of record i+1 ends at marker i+1.
# The text portion is contiguous printable ASCII. Find its start by scanning backward from marker i+1.
for i in range(len(records)):
    mpos = records[i][0]
    fstart = mpos + 5
    if i + 1 < len(records):
        nxt_marker = records[i+1][0]
        # scan backward from nxt_marker-1 while printable ascii or newline
        j = nxt_marker - 1
        while j > fstart and (32 <= content[j] <= 126 or content[j] in (10, 13)):
            j -= 1
        text_start = j + 1
        flen = text_start - fstart
    else:
        # last record: floats go to end (maybe trailing 0xa)
        end = len(content)
        while end > fstart and (content[end-1] == 10 or content[end-1] == 13 or content[end-1] == 0):
            end -= 1
        text_start = end
        flen = end - fstart
    records[i] = (mpos, records[i][1], fstart, flen, text_start)

with open(r'd:\git\ka_zhi_shen\structure.txt', 'w', encoding='utf-8') as out:
    for i, (mpos, desc, fstart, flen, text_start) in enumerate(records):
        # The description includes previous float data as non-printable; show only printable part
        d = desc.decode('ascii', errors='replace').replace('\n', ' ')
        # strip leading garbage: keep from first letter
        out.write(f'IDX {i}: bin_start={fstart} float_len={flen/4:.1f} floats({flen//4}) desc_start={text_start}\n')
        out.write(f'  DESC: {d[-250:]}\n')
print('done, records:', len(records))
