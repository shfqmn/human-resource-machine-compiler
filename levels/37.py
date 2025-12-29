allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles: list = [
    "E", 13, None, "C", 23, 
    None, None, None, None, None, 
    "P", 20, None, "S", 3, 
    None, None, None, None, None, 
    "E", -1,None, "A", 10,
    ]

def main():
    while True:
        a = int(input())
        while True:
            b = tiles[a]
            c = int(tiles[a+1])
            print(b)
            if c == -1:
                break
            else:
                a = c





if __name__ == "__main__":
    main()
