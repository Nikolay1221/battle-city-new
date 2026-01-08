import os

INPUT_ROM = 'BattleCity (Japan).nes'
OUTPUT_ROM = 'BattleCity_fixed.nes'

if not os.path.exists(INPUT_ROM):
    print(f"❌ Файл {INPUT_ROM} не найден.")
    exit()

with open(INPUT_ROM, 'rb') as f:
    data = bytearray(f.read())

print(f"Original Header (hex): {data[:16].hex()}")

# iNES header is 16 bytes.
# Bytes 0-3: 'NES\x1a'
# Bytes 11-15 MUST be 0 for nes-py to accept it as specific iNES 1.0 format.
# Many ROMs have garbage here or NES 2.0 data that nes-py doesn't understand.

fixed_count = 0
for i in range(11, 16):
    if data[i] != 0:
        data[i] = 0
        fixed_count += 1

with open(OUTPUT_ROM, 'wb') as f:
    f.write(data)

print(f"Fixed Header    (hex): {data[:16].hex()}")
print(f"✅ Создан исправленный файл: {OUTPUT_ROM}")
print(f"Исправлено байт: {fixed_count}")
