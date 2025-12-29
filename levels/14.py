allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "JUMP", "JUMPZ", "JUMPN"]
tiles = 3

def main():
    while True:
        a = int(input())
        b = int(input())
        if a > b:
            print(a)
        else:
            print(b)


if __name__ == "__main__":
    main()
