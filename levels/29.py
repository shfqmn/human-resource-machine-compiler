allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    "N","K","A","E","R",
    "D","O","L","Y","J",
    None,None,8,None,None
    ]

def main():
    while True:
        a = int(input())
        b = tiles[a]
        print(b)



if __name__ == "__main__":
    main()
