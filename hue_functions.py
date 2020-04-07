import requests
import json


def get_light_state(light):
    """
    Get state information about specific light
    :param light: str, light url
    :return: state: dict, information about light's state
    """
    r = requests.get(light)
    info = json.loads(r.text)
    state = info['state']
    return state


def get_connected_lights(bridge_url, user):
    """
    Get a list of light urls connected to bridge
    :param bridge_url: str, bridge url
    :param user: str, user id
    :return: lights, list, url for each light connected
    """
    lights_info = requests.get(bridge_url+'/api/'+user+'/lights')
    light_ids = list(json.loads(lights_info.text).keys())
    lights = [bridge_url+'/api/'+user+'/lights/'+idx for idx in light_ids]
    return lights


def set_brightness(light, value):
    """
    Sets the brightness value for a single light
    :param light: str, light url
    :param value: int, brightness value between 0 and 254
    :return: None
    """
    light_url = light + '/state'
    value = max(min(value, 254), 0)
    requests.put(light_url, data=b'{"bri":%d}' % value)


def brighter(light, increment=40):
    """
    Increases brightness by given increment.
    :param light: str, light url
    :param increment: int, increment value
    :return: None
    """
    info = json.loads(requests.get(light).text)
    brightness = info['state']['bri']
    light_url = light + '/state'
    new_value = brightness + increment
    new_value = max(min(new_value, 254), 0)
    requests.put(light_url, data=b'{"bri":%d}' % new_value)


def darker(light, increment=40):
    """
    Decreases brightness by given increment.
    :param light: str, light url
    :param increment: int, increment value
    :return: None
    """
    info = json.loads(requests.get(light).text)
    brightness = info['state']['bri']
    light_url = light + '/state'
    new_value = brightness - increment
    new_value = max(min(new_value, 254), 0)
    requests.put(light_url, data=b'{"bri":%d}' % new_value)


def turn_on(light):
    """
    Turns light on
    :param light: str, light url
    :return: None
    """
    light_url = light + '/state'
    requests.put(light_url, data=b'{"on":true}')


def turn_off(light):
    """
    Turns light off
    :param light: str, light url
    :return: None
    """
    light_url = light + '/state'
    requests.put(light_url, data=b'{"on":false}')


def set_color(light, hue, sat=None):
    """
    Sets specific color for given light
    :param light: light url
    :param hue: int, hue value between 0 and 65535
    :param sat: int, saturation value between 0 and 254
    :return:
    """
    light_url = light + '/state'
    if sat is None:
        requests.put(light_url, data=b'{"hue":%d}' % hue)
    else:
        requests.put(light_url, data=b'{"hue":%d, "sat":%d}' % (hue, sat))


def set_color_all(bridge_url, user, hue, sat=None):
    """
    Sets specific color for all lights connected to bridge
    :param bridge_url: str, bridge url
    :param user: str, user id
    :param hue: int, hue value
    :param sat: int, saturation value
    :return: None
    """
    lights_info = requests.get(bridge_url + '/api/' + user + '/lights')
    light_ids = list(json.loads(lights_info.text).keys())
    for idx in light_ids:
        light = bridge_url + '/api/' + user + '/lights/' + idx
        set_color(light, hue, sat)