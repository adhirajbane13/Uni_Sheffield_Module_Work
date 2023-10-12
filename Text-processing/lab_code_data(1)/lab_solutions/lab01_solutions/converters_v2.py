

def convert_centigrade_to_fahrenheit(centigrade):
    fahrenheit = centigrade * 9.0 / 5.0 + 32
    return fahrenheit

def convert_centigrade_to_kelvin(centigrade):
    kelvin = centigrade + 273.15
    return kelvin

def convert_distance(miles):
    kilometers = (miles * 8.0) / 5.0
    print("Converting distance in miles to kilometers:")
    print("Distance in miles:     ", miles)
    print("Distance in kilometers:", kilometers)
    return kilometers

