import re

# 변환할 파일 경로
input_file = 'requirements.txt'
output_file = 'compatible_requirements.txt'

# 파일을 읽고 변환
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # "@" 문자를 기준으로 앞의 패키지명만 추출
        cleaned_line = re.split(r'\s*@\s*', line)[0]
        outfile.write(cleaned_line + '\n')

print(f"Cleaned requirements have been saved to {output_file}")
