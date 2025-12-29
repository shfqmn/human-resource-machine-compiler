allowed = ["INBOX","OUTBOX", "COPYFROM", "COPYTO", "ADD", "SUB", "BUMPUP", "BUMPDN", "JUMP", "JUMPZ", "JUMPN"]
tiles = [
    None, None, None, None, None,
    None, None, None, None, None,
    None, None, None, None, None,
    None, None, None, None, None,
    None, None, None, None, 0
    ]

def main():
    while True:
        n = int(input())
        divisor = 2
        while divisor * divisor <= n:
            while True:
                # Division: n / divisor
                quotient = 0
                remainder = n
                while remainder >= divisor:
                    remainder -= divisor
                    quotient += 1
                
                if remainder == 0:
                    print(divisor)
                    n = quotient
                else:
                    break
            divisor += 1
        
        if n > 1:
            print(n)

if __name__ == "__main__":
    main()
