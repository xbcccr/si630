import os,sys,re

def main():

        if len(sys.argv) != 3:
                print("usage: python scorer.py trial-pages.emails.txt webpages.emails.txt")



        with open(sys.argv[1]) as f:
                gold_lines = f.readlines()

        with open(sys.argv[2]) as f:
                test_lines = f.readlines()

        if len(gold_lines) != len(test_lines):
                print("Expected same number of lines in each but %d != %d" % (len(gold_lines), len(test_lines)))
                return

        matches = 0
        for g, t in zip(gold_lines, test_lines):
                if g == t:
                        matches += 1

        print("Score: %f (%d/%d)" % (float(matches) /len(gold_lines), matches, len(gold_lines)))

if __name__ == '__main__':
        main()
