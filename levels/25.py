allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None,
    None, None, 0
]

def main():
    while True:
        a = int(input())
        total = 0
        while a > 0:
            total += a
            a -= 1
        print(total)

if __name__ == "__main__":
    main()
