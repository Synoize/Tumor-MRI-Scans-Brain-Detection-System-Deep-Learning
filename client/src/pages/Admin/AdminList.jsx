import React, { useEffect } from 'react'
import { useAppContext } from '../../context/AppContext';
import { MaterialReactTable } from "material-react-table";
import { Link } from "react-router-dom";

const columns = [
  { accessorKey: "_id", header: "Admin ID" },
  { accessorKey: "name", header: "Full Name" },
  { accessorKey: "phone", header: "Phone Number" },
  { accessorKey: "email", header: "Email Id" },
  { accessorKey: "createdAt", header: "Created" },
  { accessorKey: "updatedAt", header: "Updated" },
];

const AdminList = () => {
  const { adminList, setAdminList, getAllAdminsByAdmin } = useAppContext();

  useEffect(() => {
    getAllAdminsByAdmin();
  }, []);

  return (
    <div className="overflow-y-scroll h-screen w-full p-6 bg-gray-100">
      <div className="flex items-center justify-between bg-white p-3 rounded-lg shadow mb-4">
        <h2 className="text-lg font-bold text-gray-700">Admin List</h2>
        <Link to={"/add-admin"} className="bg-primary-green hover:bg-primary-green-dull text-white px-3 py-2 rounded-lg">
          Add Admin
        </Link>
      </div>
      <MaterialReactTable columns={columns} data={adminList} />
    </div>
  )
}

export default AdminList
