import { useState } from "react";
import { toast } from "react-toastify";
import { useAppContext } from "../context/AppContext";
import { assets } from "../assets/assets";
import ButtonLoader from "./ButtonLoader";

const Login = () => {
    const [state, setState] = useState("login");
    const {
        setShowUserLogin,
        setUser,
        navigate,
        axios,
        loading,
        setLoading
    } = useAppContext();

    const defaultUserData = {
        name: "",
        phone: "",
        email: "",
        password: ""
    };

    const [userData, setUserData] = useState(defaultUserData);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setUserData((prevUser) => ({
            ...prevUser,
            [name]: value
        }));
    };

    const handleFormSubmit = async (e) => {
        e.preventDefault();

        if (!userData.phone || !userData.password || (state === "register" && (!userData.name || !userData.email))) {
            toast.error("Please fill in all required fields.");
            return;
        }

        setLoading(true);

        const endpoint = state === "login" ? "/api/user/login" : "/api/user/register";
        const payload = state === "login"
            ? { phone: userData.phone, password: userData.password }
            : {
                name: userData.name,
                phone: userData.phone,
                password: userData.password,
                email: userData.email
            };

        try {
            const { data } = await axios.post(endpoint, payload);
            setLoading(false);
            if (data.success) {
                setUser(data.user);
                setShowUserLogin(false);
                navigate("/");
                toast.success(data.message || (state === "login" ? "Login successful" : "Account created"));
            } else {
                toast.error(data.message || "Something went wrong.");
            }
        } catch (error) {
            setLoading(false);
            toast.error(error?.response?.data?.message || error.message || "An error occurred.");
        }
    };

    return (
        <div
            onClick={() => setShowUserLogin(false)}
            className="fixed top-0 left-0 right-0 z-30 flex items-center justify-center text-sm text-gray-600 bg-black/50 min-h-screen"
        >
            <form
                onClick={(e) => e.stopPropagation()}
                onSubmit={handleFormSubmit}
                className="flex flex-col gap-4 p-8 py-12 w-80 sm:w-[352px] rounded-lg shadow-xl border border-gray-200 bg-white"
            >
                <img src={assets.logo} alt="tumor" className="w-32 mx-auto" />
                <hr className="text-gray-100" />
                <p className="text-2xl font-medium text-center w-full">
                    User{" "}
                    <span className="text-special-red">{state === "login" ? "Login" : "Sign Up"}</span>
                </p>

                {state === "register" && (
                    <>
                        <div className="w-full">
                            <p>Name</p>
                            <input
                                name="name"
                                value={userData.name}
                                onChange={handleInputChange}
                                placeholder="Type here"
                                className="border border-gray-200 rounded w-full p-2 mt-1 outline-gray-300 text-gray-500"
                                type="text"
                                required
                            />
                        </div>
                        <div className="w-full">
                            <p>Email</p>
                            <input
                                name="email"
                                value={userData.email}
                                onChange={handleInputChange}
                                placeholder="Type here"
                                className="border border-gray-200 rounded w-full p-2 mt-1 outline-gray-300 text-gray-500"
                                type="email"
                                required
                            />
                        </div>
                    </>
                )}

                <div className="w-full">
                    <p>Phone</p>
                    <input
                        name="phone"
                        value={userData.phone}
                        onChange={handleInputChange}
                        placeholder="Type here"
                        className="border border-gray-200 rounded w-full p-2 mt-1 outline-gray-300 text-gray-500"
                        type="text"
                        pattern="\d*"
                        required
                    />
                </div>

                <div className="w-full">
                    <p>Password</p>
                    <input
                        name="password"
                        value={userData.password}
                        onChange={handleInputChange}
                        placeholder="Type here"
                        className="border border-gray-200 rounded w-full p-2 mt-1 outline-gray-300 text-gray-500"
                        type="password"
                        required
                    />
                </div>

                <p className="text-sm text-gray-500">
                    {state === "register" ? (
                        <>
                            Already have an account?{" "}
                            <span
                                onClick={() => setState("login")}
                                className="text-special-red cursor-pointer"
                            >
                                Click here
                            </span>
                        </>
                    ) : (
                        <>
                            Create an account?{" "}
                            <span
                                onClick={() => setState("register")}
                                className="text-special-red cursor-pointer"
                            >
                                Click here
                            </span>
                        </>
                    )}
                </p>

                <button
                    type="submit"
                    disabled={loading}
                    className={`transition-all text-white w-full py-2 rounded-md cursor-pointer disabled:opacity-70 ${loading
                            ? "bg-special-red/60 cursor-not-allowed"
                            : "bg-special-red/90 hover:bg-special-red cursor-pointer"
                        }`}
                >
                    {loading
                        ? state === "register"
                            ? <ButtonLoader />
                            : <ButtonLoader />
                        : state === "register"
                            ? "Create Account"
                            : "Login"}
                </button>
                <p className="m-auto">Admin login?{" "}
                    <span
                        onClick={() => navigate("/admin")}
                        className="text-special-red cursor-pointer"
                    >
                        Click here
                    </span>
                </p>
            </form>
        </div>
    );
};

export default Login;

