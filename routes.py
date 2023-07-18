routes = {
    "public": {
        "/",
        "/token",
        "/refresh-token"
    },
    "private": {
    }
}


def get_route_tier(reqRoute: str):
    for tier, route_set in routes.items():
        if reqRoute in route_set:
            return tier

