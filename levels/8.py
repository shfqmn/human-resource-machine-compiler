allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "JUMP"]
tiles = 3

def main():
    while True:
        a = int(input())
        a = a + a + a
        print(a)


if __name__ == "__main__":
    main()
