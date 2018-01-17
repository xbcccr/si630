import re


pattern1 = r'(?!\.)[!#$%&.*+\-/=?^`{|}~\w]+(?<!\.)(\s*(@+|(\s\W*at\W*\s))\s*)+([a-zA-Z0-9-]+(\s*([.]|(\s\W*dot\W*\s))\s*)+)+[a-zA-Z0-9-]+'

# fhand = open('trial-pages.txt')
fhand = open('webpages.txt')
fout = open('email-outputs.txt','w')
count = 0

for line in fhand:
    line = line.rstrip()


    match1 = re.search(pattern1, line)


    if match1:
        email = match1.group(0)
        email = re.sub(r'(\s*(@+|(\s\W*at\W*\s))\s*)+','@',email)
        email = re.sub(r'(\s*([.]|(\s\W*dot\W*\s))\s*)+','.',email)
        pos = email.find('@')
        if pos > 64:
            continue
        fout.write (email+'\n')
        count += 1
    else:
        fout.write ('None'+'\n')


fout.close()
print (count)
