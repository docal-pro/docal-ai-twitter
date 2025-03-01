import csv


def find_affected_tweets():
    with open('results.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print("Header:", header)
        
        affected_count = 0
        for row in reader:
            created_at = row[0]
            text = row[1]
            context = row[3]
            
            # Remove any leading/trailing whitespace
            text = text.strip()
            context = context.strip()
            
            # Check if text starts with @ (has a username)
            if text.startswith('@'):
                # Split text into username and message
                parts = text.split(' ', 1)
                if len(parts) > 1:
                    username = parts[0]
                    message = parts[1]
                    
                    # Check if context matches the pattern
                    expected = f"@frankdegods: {message}"
                    if context == expected:
                        affected_count += 1
                        print(f"\nAffected Tweet #{affected_count}")
                        print(f"Created At: {created_at}")
                        print(f"Text: '{text}'")
                        print(f"Context: '{context}'")
                        print(f"Expected: '{expected}'")

        print(f"\nTotal affected tweets found: {affected_count}")


if __name__ == "__main__":
    find_affected_tweets() 