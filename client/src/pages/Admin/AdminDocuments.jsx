import React, { useEffect } from 'react'
import { useAppContext } from '../../context/AppContext';
import { Link } from 'react-router-dom';
import { MaterialReactTable } from 'material-react-table';
import { toast } from 'react-toastify';

const AdminDocuments = () => {
  const { axios, documentList, setDocumentList, getAllDocumentsByAdmin } = useAppContext();
  
  const columns = [
    { accessorKey: "_id", header: "Document ID" },
    {
      accessorKey: "image",
      header: "Document Image",
      Cell: ({ cell }) => (
        <img src={cell.getValue()} alt="Vehicle" width="50" height="50" />
      ),
    },
    { accessorKey: "name", header: "Document Name" },
    {
      accessorKey: "isPreview",
      header: "Preview",
      Cell: ({ cell }) => {
        const document = cell.row.original;

        return (
          <label className="relative inline-flex items-center cursor-pointer text-gray-900 gap-3">
            <input
              type="checkbox"
              checked={document.isPreview}
              onChange={() => toggleDocumentPreview(document._id, document.isPreview)}
              className="sr-only peer"
            />
             <div className="w-12 h-7 bg-slate-300 rounded-full peer peer-checked:bg-primary-green transition-colors duration-200"></div>
          <span className="dot absolute left-1 top-1 w-5 h-5 bg-white rounded-full transition-transform duration-200 ease-in-out peer-checked:translate-x-5"></span>
          </label>
        );
      },
    },
    { accessorKey: "userId", header: "User ID" },
    { accessorKey: "createdAt", header: "Created" },
    { accessorKey: "updatedAt", header: "Updated" },
  ];

  const toggleDocumentPreview = async (id, currentPreviewState) => {
    try {
      const { data } = await axios.patch('/api/document/preview', {
        id,
        isPreview: !currentPreviewState,
      });
      if (data.success) {
        getAllDocumentsByAdmin();
        toast.success(data.message);
      } else {
        toast.error(data.message);
      }
    } catch (err) {
      toast.error(err.response?.data?.message || err.message);
    }
  };

  useEffect(() => {
    getAllDocumentsByAdmin();
  }, []);

  return (
    <div className="overflow-y-scroll h-screen w-full p-6 bg-gray-100">
      <div className="flex items-center justify-between bg-white p-3 rounded-lg shadow mb-4">
        <h2 className="text-lg font-bold text-gray-700">Document List</h2>
        <Link to={"/add-vehicle"} className="bg-primary-green hover:bg-primary-green-dull text-white px-3 py-2 rounded-lg">
          Add Document
        </Link>
      </div>
      <MaterialReactTable columns={columns} data={documentList} />
    </div>
  )
}

export default AdminDocuments
