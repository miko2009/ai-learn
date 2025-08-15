
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# 1. 初始化 FastMCP 服务器
# 创建一个名为 "weather" 的服务器实例。这个名字有助于识别这套工具。
mcp = FastMCP("weather")


# weather API host
NWS_API_BASE = "https://api.weather.gov"
NWS_ALERTS_API = f"{NWS_API_BASE}/alerts/active/area"
NWS_FORECAST_API = f"{NWS_API_BASE}/points"
# http request AGENT
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"  
    }
    async with httpx.AsyncClient() as client:
        try:
            # timeout 30
            response = await client.get(url, headers=headers, timeout=30.0)
            return response.json()
        except Exception:
            return None
        

def format_alert(feature: dict) -> str:
   
    props = feature["properties"]
    return f"""
事件: {props.get('event', '未知')}
区域: {props.get('areaDesc', '未知')}
严重性: {props.get('severity', '未知')}
描述: {props.get('description', '无描述信息')}
指令: {props.get('instruction', '无具体指令')}
"""


# --- MCP 工具定义 ---

@mcp.tool()
async def get_alerts(state: str) -> str:
    url = f"{NWS_ALERTS_API}/{state}"
    data = await make_nws_request(url)
    alerts = [format_alert(feature) for feature in data["features"]]

    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:

    points_url = f"{NWS_FORECAST_API}/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "无法获取该地点的预报数据。"

    # 第二步：从上一步的响应中提取实际的天气预报接口 URL
    forecast_url = points_data["properties"]["forecast"]
    # 第三步：请求详细的天气预报数据
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "无法获取详细的预报信息。"

    # 提取预报周期数据
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    # 遍历接下来的5个预报周期（例如：今天下午、今晚、明天...）
    for period in periods[:5]:
        forecast = f"""
{period['name']}:
温度: {period['temperature']}°{period['temperatureUnit']}
风力: {period['windSpeed']} {period['windDirection']}
预报: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    # 将格式化后的预报信息连接成一个字符串并返回
    return "\n---\n".join(forecasts)

def main():
    print("Hello from mcp-sever!")
    mcp.run(transport='stdio')
    
if __name__ == "__main__":
    main()
