allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None,
    None, None,
    None, None,
    None, None,
    None, 0
]

def main():
    while True:
        a = int(input())
        b = int(input())
        c = tiles[9]

        if b == 0:
            print(b)
        elif a == 0:
            print(a)
        else:
            while a >= b:
                a = a - b
                c = c + 1
            print(c)



if __name__ == "__main__":
    main()
