import React from 'react'
import { useAppContext } from '../context/AppContext';
import { toast } from 'react-toastify';

const ConfirmLogout = ({ item }) => {
    const { axios, setUser, navigate } = useAppContext();

    const logout = async () => {
        try {
            const { data } = await axios.get('/api/user/logout');
            if (data.success) {
                toast.success(data.message || "Logout successful");
                setUser(null);
                item.setShowConfirm(false);
                navigate("/");
            }
            else {
                toast.error(data.message || "Logout failed");
            }
        } catch (error) {
            toast.error(error.message || "Logout failed");
        }
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#00000066]">
            <div className="bg-white rounded-xl p-6 shadow-xl m-6 w-full max-w-md">
                <h2 className="text-lg font-semibold mb-4">Confirm Logout</h2>
                <p className="mb-6">Are you sure you want to Logout this Account?</p>
                <div className="flex justify-end space-x-4">
                    <button
                        onClick={() => item.setShowConfirm(false)}
                        className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 cursor-pointer"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={logout}
                        className="px-4 py-2 bg-special-red/90 hover:bg-special-red text-white rounded cursor-pointer"
                    >
                        Logout
                    </button>
                </div>
            </div>
        </div>
    )
}

export default ConfirmLogout