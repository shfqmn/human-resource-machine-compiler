allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "JUMP", "JUMPZ"]
tiles = 3

def main():
    while True:
        a = int(input())
        b = int(input())
        if a == b:
            print(a)


if __name__ == "__main__":
    main()
