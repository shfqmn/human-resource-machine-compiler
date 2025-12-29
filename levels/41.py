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
        # Read a zero-terminated string from the INBOX
        length = 0
        while True:
            val = int(input())
            if val == 0:
                break
            tiles[length] = val
            length += 1
        
        if length == 0:
            continue
            
        # Selection Sort: smallest first, biggest last
        i = 0
        while i < length:
            min_idx = i
            j = i + 1
            while j < length:
                # Compare tiles[j] and tiles[min_idx]
                # We use variables because the compiler doesn't support Subscript as RHS
                vj = tiles[j]
                vm = tiles[min_idx]
                if vj < vm:
                    min_idx = j
                j += 1
            
            # Swap tiles[i] and tiles[min_idx] if they are different
            if min_idx != i:
                temp = tiles[i]
                tiles[i] = tiles[min_idx]
                tiles[min_idx] = temp
            
            i += 1
            
        # Output the sorted string to the OUTBOX
        k = 0
        while k < length:
            print(tiles[k])
            k += 1

if __name__ == "__main__":
    main()
