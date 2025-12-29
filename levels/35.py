allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None, None, None,
    None, None, None, None, None,
    None, None, None, None, 0,
    ]

def main():
    i = 0
    while True:
        ch = input()

        j = 0
        found = 0
        while j < i:
            if tiles[j] == ch:
                found = 1
                break
            j = j + 1

        if found == 0:
            print(ch)
            tiles[i] = ch
            i = i + 1


if __name__ == "__main__":
    main()
