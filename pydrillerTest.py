
from pydriller import Repository
total_added = total_removed = 0
for commit in Repository('https://github.com/tornadoweb/tornado',to_commit='80484e767ec2d1a1a7f6b73efeee6e6cec7bbe32').traverse_commits():
    for file in commit.modified_files:
        # Sum up added and removed lines for each file in each commit
        total_added += file.added_lines
        total_removed += file.deleted_lines
print(total_added - total_removed)




