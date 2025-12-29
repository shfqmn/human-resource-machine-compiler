allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None, None, None,
    None, None, None, None, None,
    None, None, None, None, 0
    ]

def main():
    # Reverse each zero-terminated sequence from the inbox.
    while True:
        bufc = 0
        ch = input()

        # Collect until sentinel 0 (numeric zero)
        while ch != "0":
            tiles[bufc] = ch
            bufc = bufc + 1
            ch = input()
        bufc = bufc - 1

        # Emit in reverse order
        while bufc >= 0:
            val = tiles[bufc]
            print(val)
            bufc = bufc - 1

if __name__ == "__main__":
    main()
