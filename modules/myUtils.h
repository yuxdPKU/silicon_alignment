#include <cmath>
#include <iostream>

constexpr float PI = 3.14159265358979323846f;
constexpr float Deg2Rad = PI / 180.f;

constexpr int NSectors = 18;
constexpr float SectorSpanDeg = 360. / NSectors;
constexpr float SectorSpanRad = SectorSpanDeg * Deg2Rad;

// conversion from B(kGaus) to curvature for 1GeV pt
constexpr float B2C = -0.299792458e-3;

template <typename T>
int angle2Sector(T phi) {
    // Convert angle to sector ID, phi can be either in 0:2pi or -pi:pi convention
    int sect = phi * 180 / PI / SectorSpanDeg;
    if (phi < 0.f) {
        sect += NSectors - 1;
    }
    return sect;
}

template <typename T>
T sector2Angle(int sect) {
    // Convert sector to its angle center, in -pi:pi convention
    T ang = SectorSpanRad * (0.5f + sect);
    // Bring angle to range -pi:pi
    if (ang > PI) {
        ang -= 2 * PI;
    }
    return ang;
}

template <typename T>
T angle2Alpha(T phi) {
    // Convert angle to its sector alpha
    return sector2Angle<T>(angle2Sector<T>(phi));
}

template <typename T>
void sincos(T angle, T& sin, T& cos) {
    sin = std::sin(angle);
    cos = std::cos(angle);
}

template <typename T>
void rotateZ(std::array<T, 3>& xy, T alpha) {
    T sinAlpha, cosAlpha;
    sincos(alpha, sinAlpha, cosAlpha);

    const T x = xy[0];
    xy[0] = x * cosAlpha - xy[1] * sinAlpha;
    xy[1] = x * sinAlpha + xy[1] * cosAlpha;
}
