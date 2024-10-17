namespace FastDouble {
    inline DEVICE float2 quickTwoSum(float a, float b) {
        float s = a+b;
        float e = b-(s-a);
        return make_float2(s, e);
    }
    inline DEVICE float2 twoSum(float a, float b) {
        float s = a+b;
        float v = s-a;
        float e = (a-(s-v))+(b-v);
        return make_float2(s, e);
    }
    inline DEVICE float2 df64add(float2 a, float2 b) {
        float2 s = twoSum(a.x, b.x);
        float2 t = twoSum(a.y, b.y);
        s.y += t.x;
        s = quickTwoSum(s.x, s.y);
        s.y += t.y;
        return quickTwoSum(s.x, s.y);
    }
    inline DEVICE float2 df64diff(float2 a, float2 b) {
        float2 s = twoSum(a.x, -b.x);
        float2 t = twoSum(a.y, -b.y);
        s.y += t.x;
        s = quickTwoSum(s.x, s.y);
        s.y += t.y;
        return quickTwoSum(s.x, s.y);
    }
    inline DEVICE float2 twoProd(float a, float b) {
        float x = a*b;
        float y = fmaf(a, b, -x);
        return make_float2(x, y);
    }
    inline DEVICE float2 df64mult(float2 a, float2 b) {
        float2 p = twoProd(a.x, b.x);
        p.y += a.x*b.y;
        p.y += a.y*b.x;
        return quickTwoSum(p.x, p.y);
    }
    inline DEVICE float2 df64sqr(float2 a) {
        float2 p = twoProd(a.x, a.x);
        p.y += 2*a.x*a.y;
        return quickTwoSum(p.x, p.y);
    }
    inline DEVICE float2 df64div(float2 b, float2 a) {
        float xn = 1.0f/a.x;
        float yn = b.x*xn;
        float diff = (df64diff(b, df64mult(a, make_float2(yn, 0)))).x;
        float2 prod = twoProd(xn, diff);
        return df64add(make_float2(yn, 0), prod);
    }
    inline DEVICE float2 df64sqrt(float2 a) {
        float xn = rsqrtf(a.x);
        float yn = a.x*xn;
        float diff = (df64diff(a, make_float2(yn*yn, 0))).x;
        float2 prod = twoProd(xn, diff) / 2;
        return df64add(make_float2(yn, 0), prod);
    }
    inline DEVICE bool df64eq(float2 a, float2 b) {
        return a.x == b.x && a.y == b.y;
    }
    inline DEVICE bool df64neq(float2 a, float2 b) {
        return a.x != b.x || a.y != b.y;
    }
    inline DEVICE bool df64lt(float2 a, float2 b) {
        return (a.x < b.x || (a.x == b.x && a.y < b.y));
    }
    inline DEVICE float2 df64exp(float2 a) {
        const float thresh = 1.0e-20f*exp(a.x);
        float2 s = df64add(make_float2(1.0f, 0.0f), a);
        float2 p = df64sqr(a);
        float m = 2.0f;
        float2 f = make_float2(2.0f, 0.0f);
        float2 t = p/2.0f;
        while (abs(t.x) > thresh) {
            s = df64add(s, t);
            p = df64mult(p, a);
            m += 1.0f;
            f = df64mult(f, make_float2(m, 0.0f));
            t = df64div(p, f);
        }
        return df64add(s, t);
    }
    inline DEVICE float2 df64log(float2 a) {
        float2 xi = make_float2(0.0f, 0.0f);
        if (!df64eq(a, make_float2(1.0f, 0.0f))) {
            if (a.x <= 0.0f)
                xi = make_float2(logf(a.x));
            else {
                xi.x = log(a.x);
                xi = df64add(df64add(xi, df64mult(df64exp(-xi), a)), make_float2(-1.0f, 0.0f));
            }
        }
        return xi;
    }
    inline DEVICE float2 df64sin(float2 a) {
        const float thresh = 1.0e-20f*abs(a.x);
        if (a.x == 0.0f)
            return make_float2(0.0f, 0.0f);
        float2 x = -df64sqr(a);
        float2 s = a;
        float2 p = a;
        float m = 1.0f;
        float2 f = make_float2(1.0f, 0.0f);
        while (true) {
            p = df64mult(p, x);
            m += 2.0f;
            f = df64mult(f, make_float2(m*(m-1), 0.0f));
            float2 t = df64div(p, f);
            s = df64add(s, t);
            if (abs(t.x) < thresh)
                break;
        }
        return s;
    }
    inline DEVICE float2 df64cos(float2 a) {
        const float thresh = 1.0e-20f*abs(a.x);
        const float2 one = make_float2(1.0f, 0.0f);
        if (a.x == 0.0f)
            return one;
        float2 x = -df64sqr(a);
        float2 c = one;
        float2 p = one;
        float m = 0.0f;
        float2 f = one;
        while (true) {
            p = df64mult(p, x);
            m += 2.0f;
            f = df64mult(f, make_float2(m*(m-1), 0.0f));
            float2 t = df64div(p, f);
            c = df64add(c, t);
            if (abs(t.x) < thresh)
                break;
        }
        return c;
    }
    const double SPLITTER = (1<<29)+1;
}

class fastdouble {
public:
    float2 v;
    DEVICE fastdouble(double d) {
        double t = d*FastDouble::SPLITTER;
        double t_hi = t-(t-d);
        v = make_float2((float) t_hi, (float) (d-t_hi));
    }
    DEVICE fastdouble(float f) {
        v = make_float2(f, 0.0f);
    }
    DEVICE fastdouble(int i) {
        v = make_float2(i, 0.0f);
    }
    DEVICE fastdouble(float2 f) {
        v = f;
    }
    DEVICE operator float() const {
        return v.x;
    }
    DEVICE operator double() const {
        return (double) v.x + (double) v.y;
    }
    DEVICE bool operator==(const fastdouble& rhs) const {
        return FastDouble::df64eq(v, rhs.v);
    }
    DEVICE bool operator!=(const fastdouble& rhs) const {
        return FastDouble::df64neq(v, rhs.v);
    }
    DEVICE bool operator<(const fastdouble& rhs) const {
        return FastDouble::df64lt(v, rhs.v);
    }
    DEVICE bool operator>(const fastdouble& rhs) const {
        return rhs < *this;
    }
    DEVICE bool operator<=(const fastdouble& rhs) const {
        return !(*this > rhs);
    }
    DEVICE bool operator>=(const fastdouble& rhs) const {
        return !(*this < rhs);
    }
    DEVICE fastdouble operator-() const {
        return fastdouble(make_float2(-v.x, -v.y));
    }
    DEVICE fastdouble operator+(const fastdouble& rhs) const {
        return fastdouble(FastDouble::df64add(v, rhs.v));
    }
    DEVICE fastdouble operator-(const fastdouble& rhs) const {
        return fastdouble(FastDouble::df64diff(v, rhs.v));
    }
    DEVICE fastdouble operator*(const fastdouble& rhs) const {
        return fastdouble(FastDouble::df64mult(v, rhs.v));
    }
    DEVICE fastdouble operator/(const fastdouble& rhs) const {
        return fastdouble(FastDouble::df64div(v, rhs.v));
    }
};

inline DEVICE fastdouble operator+(fastdouble a, double b) {
    return a+fastdouble(b);
}

inline DEVICE fastdouble operator+(fastdouble a, float b) {
    return a+fastdouble(b);
}

inline DEVICE fastdouble operator+(fastdouble a, int b) {
    return a+fastdouble(b);
}

inline DEVICE fastdouble operator+(double a, fastdouble b) {
    return fastdouble(a)+b;
}

inline DEVICE fastdouble operator+(float a, fastdouble b) {
    return fastdouble(a)+b;
}

inline DEVICE fastdouble operator+(int a, fastdouble b) {
    return fastdouble(a)+b;
}

inline DEVICE fastdouble operator-(fastdouble a, double b) {
    return a-fastdouble(b);
}

inline DEVICE fastdouble operator-(fastdouble a, float b) {
    return a-fastdouble(b);
}

inline DEVICE fastdouble operator-(fastdouble a, int b) {
    return a-fastdouble(b);
}

inline DEVICE fastdouble operator-(double a, fastdouble b) {
    return fastdouble(a)-b;
}

inline DEVICE fastdouble operator-(float a, fastdouble b) {
    return fastdouble(a)-b;
}

inline DEVICE fastdouble operator-(int a, fastdouble b) {
    return fastdouble(a)-b;
}

inline DEVICE fastdouble operator*(fastdouble a, double b) {
    return a*fastdouble(b);
}

inline DEVICE fastdouble operator*(fastdouble a, float b) {
    return a*fastdouble(b);
}

inline DEVICE fastdouble operator*(fastdouble a, int b) {
    return a*fastdouble(b);
}

inline DEVICE fastdouble operator*(double a, fastdouble b) {
    return fastdouble(a)*b;
}

inline DEVICE fastdouble operator*(float a, fastdouble b) {
    return fastdouble(a)*b;
}

inline DEVICE fastdouble operator*(int a, fastdouble b) {
    return fastdouble(a)*b;
}

inline DEVICE fastdouble operator/(fastdouble a, double b) {
    return a/fastdouble(b);
}

inline DEVICE fastdouble operator/(fastdouble a, float b) {
    return a/fastdouble(b);
}

inline DEVICE fastdouble operator/(fastdouble a, int b) {
    return a/fastdouble(b);
}

inline DEVICE fastdouble operator/(double a, fastdouble b) {
    return fastdouble(a)/b;
}

inline DEVICE fastdouble operator/(float a, fastdouble b) {
    return fastdouble(a)/b;
}

inline DEVICE fastdouble operator/(int a, fastdouble b) {
    return fastdouble(a)/b;
}

inline DEVICE fastdouble sqrt(fastdouble a) {
    return fastdouble(FastDouble::df64sqrt(a.v));
}

inline DEVICE fastdouble log(fastdouble a) {
    return fastdouble(FastDouble::df64log(a.v));
}

inline DEVICE fastdouble exp(fastdouble a) {
    return fastdouble(FastDouble::df64exp(a.v));
}

inline DEVICE fastdouble sin(fastdouble a) {
    return fastdouble(FastDouble::df64sin(a.v));
}

inline DEVICE fastdouble cos(fastdouble a) {
    return fastdouble(FastDouble::df64cos(a.v));
}

inline DEVICE fastdouble tan(fastdouble a) {
    float2 s = FastDouble::df64sin(a.v);
    float2 c = FastDouble::df64cos(a.v);
    return fastdouble(s)/fastdouble(c);
}
