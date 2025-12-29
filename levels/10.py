allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "JUMP"]
tiles = 5

def main():
    while True:
        a = int(input())
        a2 = a + a
        a4 = a2 + a2
        a8 = a4 + a4
        print(a8)


if __name__ == "__main__":
    main()
