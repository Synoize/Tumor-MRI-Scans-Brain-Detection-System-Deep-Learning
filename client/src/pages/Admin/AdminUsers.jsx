import React, { useEffect } from 'react'
import { useAppContext } from '../../context/AppContext';
import { MaterialReactTable } from "material-react-table";
import { Link } from "react-router-dom";
import { toast } from 'react-toastify';

const columns = [
  { accessorKey: "_id", header: "User ID" },
  { accessorKey: "name", header: "Full Name" },
  { accessorKey: "phone", header: "Phone Number" },
  { accessorKey: "email", header: "Email Id" },
  {
    accessorKey: "image",
    header: "Document Image",
    Cell: ({ cell }) => (
      <img src={cell.getValue()} alt="Vehicle" width="50" height="50" />
    ),
  },
  {
    accessorKey: "uploadlimit",
    header: "Upload Limit",
    Cell: ({ row, cell }) => {

      const {axios, authUser} = useAppContext();
      const [isEditing, setIsEditing] = React.useState(false);
      const [newLimit, setNewLimit] = React.useState(cell.getValue());

      const handleUpdate = async () => {
        try {
          const userId = row.original._id;

          const res = await axios.patch('/api/admin/upgrade-plan', {
            id: userId,
            uploadlimit: Number(newLimit)
          });

          if (res.data.success) {
            authUser();
            toast.success(res.data.message);
            setIsEditing(false);
          } else {
            toast.error(res.data.message);
          }
        } catch (error) {
          toast.error("Failed to update upload limit");
        }
      };

      return (
        <div className="flex items-center gap-2">
          {isEditing ? (
            <>
              <input
                type="number"
                value={newLimit}
                onChange={(e) => setNewLimit(e.target.value)}
                className="w-14 px-2 py-1 border rounded"
              />
              <button
                onClick={handleUpdate}
                className="text-sm px-2 py-1 bg-primary-green text-white rounded hover:bg-primary-green-dull"
              >
                Save
              </button>
            </>
          ) : (
            <>
              <span className={`${newLimit > 0 ? 'text-black' : 'text-primary'}`}>
                {newLimit}
              </span>
              <button
                onClick={() => setIsEditing(true)}
                className="text-sm text-blue-500 underline hover:text-blue-700"
              >
                {/* <LuPencilLine/> */}
              </button>
            </>
          )}
        </div>
      );
    }
  },
  { accessorKey: "documentName", header: "Document Name" },
  { accessorKey: "createdAt", header: "Created" },
  { accessorKey: "updatedAt", header: "Updated" },
];

const AdminUsers = () => {
  const { userList, documentList, getAllUsersByAdmin } = useAppContext();
  console.log("User List: ", userList);

  useEffect(() => {
    getAllUsersByAdmin();
  }, []);

  return (
    <div className="overflow-y-scroll h-screen w-full p-6 bg-gray-100">
      <div className="flex items-center justify-between bg-white p-3 rounded-lg shadow mb-4">
        <h2 className="text-lg font-bold text-gray-700">User List</h2>
        <Link to={"/add-user"} className="bg-primary-green hover:bg-primary-green-dull text-white px-3 py-2 rounded-lg">
          Add User
        </Link>
      </div>
      <MaterialReactTable columns={columns} data={userList} />
    </div>
  )
}

export default AdminUsers
