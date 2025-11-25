import { useState } from "react";
import { useAppContext } from "../../context/AppContext";
import { assets } from "../../assets/assets";
import ButtonLoader from "../ButtonLoader";
import { toast } from "react-toastify";

const AdminLogin = () => {
    const [authState, setAuthState] = useState("login");

    const {
        setShowAdminLogin,
        setAdmin,
        navigate,
        axios,
        loading,
        setLoading
    } = useAppContext();

    const defaultAdminData = {
        name: "",
        phone: "",
        email: "",
        password: ""
    };

    const [adminData, setAdminData] = useState(defaultAdminData);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setAdminData((prev) => ({ ...prev, [name]: value }));
    };

    const handleFormSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        const endpoint = authState === "login" ? "/api/admin/login" : "/api/admin/register";
        const payload = authState === "login"
            ? {
                phone: adminData.phone,
                password: adminData.password
            }
            : {
                name: adminData.name,
                phone: adminData.phone,
                email: adminData.email,
                password: adminData.password
            };

        try {
            const { data } = await axios.post(endpoint, payload);
            if (data.success) {
                setAdmin(data.admin);
                setShowAdminLogin(false);
                navigate("/admin");

                toast.success(data.message || (authState === "login" ? "Login successful" : "Account created"));
            } else {
                toast.error(data.message || "Something went wrong");
            }
        } catch (error) {
            const message = error.response?.data?.message || error.message || "An error occurred";
            toast.error(message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed top-0 left-0 right-0 z-30 flex items-center justify-center text-sm text-gray-600 bg-black/50 min-h-screen">
            <form
                onClick={(e) => e.stopPropagation()}
                onSubmit={handleFormSubmit}
                className="flex flex-col gap-4 p-8 py-12 w-80 sm:w-[352px] rounded-lg shadow-xl border border-gray-200 bg-white"
            >
                <img src={assets.logo} alt="tumor" className="w-32 mx-auto" />
                <hr className="text-gray-100" />
                <p className="text-2xl font-medium text-center w-full">
                    Admin{" "}
                    <span className="text-special-red">{authState === "login" ? "Login" : "Sign Up"}</span>
                </p>

                {authState === "register" && (
                    <>
                        <div className="w-full">
                            <p>Name</p>
                            <input
                                name="name"
                                value={adminData.name}
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
                                value={adminData.email}
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
                        value={adminData.phone}
                        onChange={handleInputChange}
                        placeholder="Type here"
                        className="border border-gray-200 rounded w-full p-2 mt-1 outline-gray-300 text-gray-500"
                        type="tel"
                        required
                    />
                </div>

                <div className="w-full">
                    <p>Password</p>
                    <input
                        name="password"
                        value={adminData.password}
                        onChange={handleInputChange}
                        placeholder="Type here"
                        className="border border-gray-200 rounded w-full p-2 mt-1 outline-gray-300 text-gray-500"
                        type="password"
                        required
                    />
                </div>

                <p className="text-sm text-gray-500">
                    {authState === "register" ? (
                        <>
                            Already have an account?{" "}
                            <span
                                onClick={() => setAuthState("login")}
                                className="text-special-red cursor-pointer"
                            >
                                Click here
                            </span>
                        </>
                    ) : (
                        <>
                            Create an account?{" "}
                            <span
                                onClick={() => setAuthState("register")}
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
                    className={`transition-all text-white w-full py-2 rounded-md cursor-pointer ${loading
                            ? "bg-special-red/60 cursor-not-allowed"
                            : "bg-special-red/90 hover:bg-special-red cursor-pointer"
                        }`}
                >
                    {loading
                        ? authState === "register"
                            ? <ButtonLoader />
                            : <ButtonLoader />
                        : authState === "register"
                            ? "Create Account"
                            : "Login"}
                </button>

                <p className="m-auto">User login?{" "}
                    <span
                        onClick={() => navigate("/")}
                        className="text-special-red cursor-pointer"
                    >
                        Click here
                    </span>
                </p>
            </form>
        </div>
    );
};

export default AdminLogin;

