import requests

base_url = 'http://r2d2.hackingarena.com:1811/song.php?songid='

for i in range(1, 10001):  # range from 1 to 10000
    url = base_url + str(i)
    response = requests.get(url)
    
    content = response.content.decode('utf-8').strip()
    if content != '<h1>':
        print(f'id: {i}, Content: {content}')
