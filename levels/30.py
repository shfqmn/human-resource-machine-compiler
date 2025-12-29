allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    'G','E','T',0,'T',
    'H',0,'T','A','R',
    0,'A','W','A','K',
    'E',0,'I','S','0',
    'X','X','X',0,None,
    ]
modifiable_tiles_idx = [23,24]

def main():
    while True:
        a = int(input())
        while tiles[a] != 0:
            print(tiles[a])
            a = a + 1



if __name__ == "__main__":
    main()
