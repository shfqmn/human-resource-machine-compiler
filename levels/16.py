allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "JUMP", "JUMPZ", "JUMPN"]
tiles = 3

def main():
    while True:
        a = int(input())
        if a < 0:
            a = a - a - a
        print(a)



if __name__ == "__main__":
    main()
