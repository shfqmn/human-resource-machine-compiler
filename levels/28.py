allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = 10

def main():
    while True:
        a = int(input())
        b = int(input())
        c = int(input())

        if a > b:
            if b > c:
                print(c)
                print(b)
                print(a)
            elif a > c:
                print(b)
                print(c)
                print(a)
            else:
                print(b)
                print(a)
                print(c)
        else:
            if a > c:
                print(c)
                print(a)
                print(b)
            elif b > c:
                print(a)
                print(c)
                print(b)
            else:
                print(a)
                print(b)
                print(c)



if __name__ == "__main__":
    main()
