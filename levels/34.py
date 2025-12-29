allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    "A", "E", "I", "O", "U",
    0 , None, None, None, None
    ]
modifiable_tiles_idx = [5,6,7,8,9]

def main():
    while True:
        ch = input()
        if ch != "A" and ch != "E" and ch != "I" and ch != "O" and ch != "U":
            print(ch)

if __name__ == "__main__":
    main()
