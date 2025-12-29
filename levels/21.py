allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None, 
    None, None, 0
    ]

def main():
    while True:
        total = 0
        while True:
            a = int(input())
            if a == 0:
                break
            total += a
        print(total)

if __name__ == "__main__":
    main()
