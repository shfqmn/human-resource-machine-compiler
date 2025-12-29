allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles: list = [
    None, None, None,
    None, None, None,
    None, None, None,
    0, 10, 100,
    ]

def main():
    while True:
        a = int(input())

        if a < 10:
            print(a)
        elif a < 100:
            acc = 0
            while True:
                a = a - 10
                acc = acc + 1
                if a < 10:
                    break
            print(acc)
            print(a)
        else:
            acc = 0
            while True:
                a = a - 100
                acc = acc + 1
                if a < 100:
                    break
            print(acc)

            acc = 0
            while a >= 10:
                a = a - 10
                acc = acc + 1
            print(acc)
            print(a)

if __name__ == "__main__":
    main()
