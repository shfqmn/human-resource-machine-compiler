allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None,
    None, None,
    0, 1,
    ]

def main():
    while True:
        a = int(input())
        b = int(input())
        if (a > 0 and b > 0) or (a < 0 and b < 0):
            print(tiles[4])
        else:
            print(tiles[5])


if __name__ == "__main__":
    main()
