allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None, None, None, 
    None, None, None, None, 0
    ]

def main():
    while True:
        a = int(input())
        b = tiles[9]+1
        c = tiles[9]+1

        print(b)
        while b <= a:
            print(b)
            d = b
            b = b + c
            c = d

if __name__ == "__main__":
    main()
