import re

pattern1 = r'(\b\w+@\S+\b)'
pattern2 = r'\b(\w+)\s\W*at\W*\s(\w+)\s\Wdot\W\s(\w+)\b'
pattern3 = r'(?!\.)[!#$%&.*+\-/=?^`{|}~\w]+(?<!\.)(\s*(@+|(\s\W*at\W*\s))\s*)+([a-zA-Z0-9-]+(\s*([.]|(\s\W*dot\W*\s))\s*)+)+[a-zA-Z0-9-]+'

# fhand = open('trial-pages.txt')
fhand = open('webpages.txt')
fout = open('webpages.emails.txt','w')
count = 0

for line in fhand:
    line = line.rstrip()

    match1 = re.search(pattern1,line)
    match2 = re.search(pattern2,line)
    match3 = re.search(pattern3, line)


    if match3:
        email = match3.group(0)
        email = re.sub(r'(\s*(@+|(\s\W*at\W*\s))\s*)+','@',email)
        email = re.sub(r'(\s*([.]|(\s\W*dot\W*\s))\s*)+','.',email)
        fout.write (email+'\n')
        count += 1
    else:
        fout.write ('None'+'\n')

fout.close()
print (count)
