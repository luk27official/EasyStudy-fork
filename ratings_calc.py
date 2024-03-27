import csv

# open the csv
csvfile = open("server/static/datasets/ml-latest/ratings.csv", "r")

# read the csv
csvreader = csv.reader(csvfile)

# get the header
header = next(csvreader)

SEARCH_ID = "28"
log = False

# now iterate through all rows and find the ids that match
avg = 0
count = 0
for row in csvreader:
    if row[1] == SEARCH_ID:
        if log:
            print(row)
        avg += float(row[2])
        count += 1

print(avg / count)
