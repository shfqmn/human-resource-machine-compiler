allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "JUMP", "JUMPZ"]
tiles = 3

def main():
    while True:
        a = int(input())
        b = int(input())
        print(b - a)
        print(a - b)

if __name__ == "__main__":
    main()
